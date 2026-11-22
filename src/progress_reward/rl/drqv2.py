"""DrQ-v2 (Yarats et al., 2021) with the network of RARM's Table 5 and the released defaults:
feature_dim 50, hidden_dim 1024, lr 1e-4 (Adam for encoder, actor and critic), critic target
EMA 0.01, stddev schedule linear(1.0, 0.1, 500000), stddev clip 0.3, random-shift augmentation
(pad 4), uniform random actions for the first num_expl_steps (2000) agent steps.

Written from the paper and the DrQ-v2 algorithm description; not copied from either codebase.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def schedule(spec, step):
    match = re.match(r"linear\((.+),(.+),(.+)\)", spec)
    start, end, duration = (float(g) for g in match.groups())
    mix = min(max(step / duration, 0.0), 1.0)
    return (1.0 - mix) * start + mix * end


class RandomShift(nn.Module):
    """Replicate-pad by 4, then crop back to the input size at a random integer offset."""

    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        self.apply(weight_init)

    def forward(self, obs):
        return self.convnet(obs / 255.0 - 0.5).flatten(1)


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low, self.high, self.eps = low, high, eps

    def _clamp(self, x):
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        eps = torch.randn(self._extended_shape(sample_shape), dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


def trunk(repr_dim, feature_dim):
    return nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())


class Actor(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = trunk(repr_dim, feature_dim)
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(weight_init)

    def forward(self, h, std):
        mu = torch.tanh(self.policy(self.trunk(h)))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = trunk(repr_dim, feature_dim)

        def head():
            return nn.Sequential(nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.Q1, self.Q2 = head(), head()
        self.apply(weight_init)

    def forward(self, h, action):
        x = torch.cat([self.trunk(h), action], dim=-1)
        return self.Q1(x), self.Q2(x)


class DrQV2:
    def __init__(self, channels, action_dim, device, lr=1e-4, feature_dim=50, hidden_dim=1024,
                 critic_target_tau=0.01, num_expl_steps=2000, stddev_schedule="linear(1.0,0.1,500000)",
                 stddev_clip=0.3):
        self.device, self.action_dim = device, action_dim
        self.critic_target_tau, self.num_expl_steps = critic_target_tau, num_expl_steps
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip
        self.encoder = Encoder(channels).to(device)
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.aug = RandomShift(pad=4)

    @torch.no_grad()
    def act(self, obs, step, eval_mode):
        """obs uint8 [N, C, 84, 84] numpy -> actions [N, A] numpy."""
        obs = torch.as_tensor(obs, device=self.device).float()
        dist = self.actor(self.encoder(obs), schedule(self.stddev_schedule, step))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()

    def update(self, batch, step):
        obs, action, reward, discount, next_obs = batch
        obs, next_obs = self.aug(obs.float()), self.aug(next_obs.float())
        h = self.encoder(obs)
        with torch.no_grad():
            next_h = self.encoder(next_obs)
        std = schedule(self.stddev_schedule, step)

        with torch.no_grad():
            dist = self.actor(next_h, std)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_h, next_action)
            target_q = reward + discount * torch.min(target_q1, target_q2)
        q1, q2 = self.critic(h, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        h = h.detach()
        dist = self.actor(h, std)
        new_action = dist.sample(clip=self.stddev_clip)
        q1_pi, q2_pi = self.critic(h, new_action)
        actor_loss = -torch.min(q1_pi, q2_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.lerp_(p, self.critic_target_tau)
        return dict(critic_loss=float(critic_loss), actor_loss=float(actor_loss), q=float(q1.mean()),
                    target_q=float(target_q.mean()), batch_reward=float(reward.mean()), std=std)

    # ---- checkpoints ----

    def policy_state(self):
        return dict(encoder=self.encoder.state_dict(), actor=self.actor.state_dict())

    def full_state(self):
        return dict(encoder=self.encoder.state_dict(), actor=self.actor.state_dict(),
                    critic=self.critic.state_dict(), critic_target=self.critic_target.state_dict(),
                    encoder_opt=self.encoder_opt.state_dict(), actor_opt=self.actor_opt.state_dict(),
                    critic_opt=self.critic_opt.state_dict())

    def load(self, state):
        for name in ("encoder", "actor", "critic", "critic_target", "encoder_opt", "actor_opt", "critic_opt"):
            if name in state:
                getattr(self, name).load_state_dict(state[name])
