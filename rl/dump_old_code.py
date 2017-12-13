# NON_BATCHED NUMPY VERSION
# TODO finish; convert tf vars to lists and ints
def sample_action(self, obs, policy, value):
available_actions = obs.observation['available_actions'] # TODO this is NOT BATCHED - can we ? no.. we can't batch this over environments, as each env at each step will have different acts and thus args
# BUT WE CAN BATCH THE SAMPLING ITSELF!!
a_0 = sample(policy['function_id']) # TODO use gather
out_args = []
for arg_type in actions.FUNCTIONS._func_list[a_0].args:
  arg_name = ... # TODO find arg type name or index by type directly
  pi = policy[arg_name]
  a_l = [sample()]
  if len(pi.): # map-dimensional
    flat_pi = tf.reshape(pi, [-1, ])
    flat_index = sample(flat_pi)
    a_l = [flat_index % width, flat_index // height] # TODO height, width
  out_args.append(a_l)
return actions.FunctionCall(a_0, out_args)

def sample_action_independent(available_actions, policy):
a_0 = sample(policy[0][available_actions])
out_args = []
for arg_type in actions.FUNCTIONS._func_list[a_0].args:
  is_spatial, pi = policy[1][arg_type]
  if is_spatial:
    height, width = pi.shape
    flat_pi = tf.reshape(pi, [-1])
    flat_index = sample(flat_pi)
    a_l = [flat_index % width, flat_index // height]
  else:
    a_l = [sample(pi)]
  out_args.append(a_l)

return actions.FunctionCall(a_0, out_args)

def step(sampling_fn, obs, policy, value):
available_actions = obs.observation['available_actions']
policy_np, value_np = sess.run([policy, value], feed_dict=get_feed_dict(obs)) # TODO get_feed_dict
action = sampling_fn(available_actions, policy_np)
