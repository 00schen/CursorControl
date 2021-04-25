import copy
import cv2
from gaze_capture.face_processor import FaceProcessor
import numpy as np
import os
from pathlib import Path
main_dir = str(Path(__file__).resolve().parents[2])

predictor_path = os.path.join(main_dir,'gaze_capture','model_files','shape_predictor_68_face_landmarks.dat')
webcam = cv2.VideoCapture(0)
face_processor = FaceProcessor(predictor_path)

def gaze_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    path_length = 0
    agent.reset()
    o = env.reset()
    while path_length < max_path_length:
        raw_o = o['raw_obs']
        raw_obs.append(raw_o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        raw_next_o = next_o['raw_obs']
        gaze_features = None
        while gaze_features is None:
            _, frame = webcam.read()
            gaze_features = face_processor.get_gaze_features(frame)
        env_info['gaze_features'] = gaze_features

        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        raw_next_obs.append(raw_next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )