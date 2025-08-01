from abc import abstractmethod

import equinox as eqx
from jax import random as jr
from jaxtyping import Array, Bool, Float, Key

from oryx.env import AbstractEnv, AbstractEnvLike
from oryx.spaces import AbstractSpace


class AbstractWrapper[WrapperActType, WrapperObsType, ActType, ObsType](
    AbstractEnvLike[WrapperActType, WrapperObsType]
):
    """Base class for environment wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    @property
    def unwrapped(self) -> AbstractEnv[ActType, ObsType]:
        """Return the unwrapped environment"""
        return self.env.unwrapped


class AbstractObservationWrapper[WrapperObsType, ActType, ObsType](
    AbstractWrapper[ActType, WrapperObsType, ActType, ObsType]
):
    """Base class for environment observation wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key | None
    ) -> tuple[eqx.nn.State, WrapperObsType, dict]:
        if key is not None:
            env_key, wrapper_key = jr.split(key, 2)
        else:
            env_key, wrapper_key = None, None

        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=env_key)
        state, obs = self.observation(state, obs, key=wrapper_key)

        state = state.update(substate)

        return state, obs, info

    def step(self, state: eqx.nn.State, action: ActType, *, key: Key | None) -> tuple[
        eqx.nn.State,
        WrapperObsType,
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        dict,
    ]:
        if key is not None:
            env_key, wrapper_key = key.split(2)
        else:
            env_key, wrapper_key = None, None

        substate = state.substate(self.env)
        substate, obs, reward, done, truncated, info = self.env.step(
            substate, action, key=env_key
        )
        state, obs = self.observation(state, obs, key=wrapper_key)

        state = state.update(substate)

        return state, obs, reward, done, truncated, info

    @abstractmethod
    def observation(
        self, state: eqx.nn.State, obs: ObsType, *, key: Key | None
    ) -> tuple[eqx.nn.State, WrapperObsType]:
        """Transform the wrapped environment observation"""

    def render(self, state: eqx.nn.State):
        return self.env.render(state)

    def close(self):
        return self.env.close()

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space


class AbstractActionWrapper[WrapperActType, ActType, ObsType](
    AbstractWrapper[WrapperActType, ObsType, ActType, ObsType]
):
    """Base class for environment action wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key | None
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=key)
        state = state.update(substate)

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key | None
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        if key is not None:
            env_key, wrapper_key = jr.split(key, 2)
        else:
            env_key, wrapper_key = None, None

        state, transformed_action = self.action(state, action, key=wrapper_key)

        substate = state.substate(self.env)
        substate, obs, reward, done, truncated, info = self.env.step(
            substate, transformed_action, key=env_key
        )
        state = state.update(substate)

        return state, obs, reward, done, truncated, info

    @abstractmethod
    def action(
        self, state: eqx.nn.State, action: WrapperActType, *, key: Key | None
    ) -> tuple[eqx.nn.State, ActType]:
        """Transform the action to the wrapped environment"""

    def render(self, state: eqx.nn.State):
        return self.env.render(state)

    def close(self):
        return self.env.close()

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space


class AbstractRewardWrapper[ActType, ObsType](
    AbstractWrapper[ActType, ObsType, ActType, ObsType]
):
    """Base class for environment reward wrappers"""

    env: eqx.AbstractVar[AbstractEnvLike[ActType, ObsType]]

    def reset(
        self, state: eqx.nn.State, *, key: Key | None
    ) -> tuple[eqx.nn.State, ObsType, dict]:
        substate = state.substate(self.env)
        substate, obs, info = self.env.reset(substate, key=key)
        state = state.update(substate)

        return state, obs, info

    def step(
        self, state: eqx.nn.State, action: ActType, *, key: Key | None
    ) -> tuple[
        eqx.nn.State, ObsType, Float[Array, ""], Bool[Array, ""], Bool[Array, ""], dict
    ]:
        if key is not None:
            env_key, wrapper_key = jr.split(key, 2)
        else:
            env_key, wrapper_key = None, None

        substate = state.substate(self.env)
        substate, obs, reward, done, truncated, info = self.env.step(
            substate, action, key=env_key
        )
        state = state.update(substate)

        reward = self.reward(state, reward, key=wrapper_key)

        return state, obs, reward, done, truncated, info

    @abstractmethod
    def reward(
        self, state: eqx.nn.State, reward: Float[Array, ""], *, key: Key | None
    ) -> Float[Array, ""]:
        """Transform the reward from the wrapped environment"""

    def render(self, state: eqx.nn.State):
        return self.env.render(state)

    def close(self):
        return self.env.close()

    @property
    def action_space(self) -> AbstractSpace[ActType]:
        return self.env.action_space

    @property
    def observation_space(self) -> AbstractSpace[ObsType]:
        return self.env.observation_space
