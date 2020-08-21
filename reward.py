import numpy as np


## If after prioritization, we miss a failure. the reward is -1, else if we don't miss any failure and also detect failures, the reward is 1,
## If not a single failure is present, the reward is zero.
## It is naive reward function.
def simple_discrete_reward(result, sc=None):
	if result[1] > 0:
		scenario_reward = -1.0
	elif result[0] > 0:
		scenario_reward = 1.0
	else:
		scenario_reward = 0.0

	return scenario_reward


## Another tyoe of reward function.
## Result[0] represents the no of detected failures, Result[1] represents the no of missed failures and Result[2] represents the ttf.
def simple_continuous_reward(result, sc=None):
	failure_count = result[0] + result[1]

	if result[1] > 0:
		scenario_reward = -1.0  #- (3*result[1]/failure_count)
	elif result[0] > 0:
		scenario_reward = 1.0 + (3.0-3.0*result[2])
	else:
		scenario_reward = 0.0

	return scenario_reward

def APHFW_reward(result, sc=None):
	return int(result[-1]*1000)

## Returns the NAPFD * Scaling Factor as the reward.
## result[3] is the NAPFD average
def napfd_reward(result, sc=None):
	#print(result[0], result[1], result[2], result[3])
	total_failures = result[0] + result[1]
	scaling_factor = 100

	if total_failures == 0:
		return 0
	elif result[0] == 0:
		return 0
	else:
		# Apply NAPFD
		return int(result[3] * scaling_factor)


def shifted_napfd_reward(result, sc=None):
	total_failures = result[0] + result[1]
	scaling_factor = 1.0

	if total_failures == 0:
		return 0.0
	elif result[0] == 0:
		return -1.0 * scaling_factor
	elif result[3] < 0.3:
		return result[3]-0.3
	else:
		# Apply NAPFD
		return result[3] * scaling_factor

##Different Reward Functions are used for giving rewards to the Agents


## 1 if result[0] is positice else 0
def binary_positive_detection_reward(result, sc=None):
	rew = 1 if result[0] > 0 else 0
	return float(rew)


## The result is taken as input and it returns the total no of failures scheduled and detected .
def failcount(result, sc=None):
	## Result[0] refers to the total no of failures scheduled and Detected
	#print('failcount : ', result[0])
	return float(result[0])

## Result of the scheduled test cases is taken as the input and it gives a reward to the test suite.
## The time rank gives reward to the complete suite and also to the position of the test case in the suite. The relation by which the passed cases are given reward can be tweaked and experimented 
## for making a more efficient reward function

def new_timerank(result, sc):
	if result[0] == 0:
		return 0.0

	total = result[0]
	rank_idx = np.array(result[-2]) - 1

	ordered_rewards = []
	fail_index = []

	for tc in sc.testcases():
		try:
			idx = sc.scheduled_testcases.index(tc)  # Slow call
			#print(idx, rank_idx)
			if idx in rank_idx:
				ordered_rewards.append(1.0)
				fail_index.append(len(ordered_rewards)-1)
			else:
				ordered_rewards.append(0.0)
		except ValueError:
			ordered_rewards.append(0.0)  # Unscheduled test case

	ordered_rewards = np.cumsum(ordered_rewards)
	ordered_rewards[fail_index] = total
	print('new_timerank : ', len(ordered_rewards), type(ordered_rewards))
	return ordered_rewards

def newtimerank(result, sc):
	if result[0] == 0:
		return 0.0


	total = result[0]
	#print(total)

	## Detection Ranks are the positions of the failed test cases after our algorithm has prioritized them.
	## 실패할꺼라고 예상 한거중에서 실제로 실패한 TC의 index가 rank_idx인듯
	rank_idx = np.array(result[-2])-1
	#print(rank_idx)
	## no_scheduled is the total no of failed test cases scheduled by the algorithm
	## 실패할꺼라고 예상한 test case들의 sequence의 길이가 no_scheduled인듯
	no_scheduled = len(sc.scheduled_testcases)
	#print('no_scheduled : ', no_scheduled)
	rewards = np.zeros(no_scheduled)
	#print(rewards)

	## The rewards for the test cases that are failing is 1
	#rewards[rank_idx] = 1
	rewards[:rank_idx[0]] = -1
	#print(rewards)

	## The time rank gives reward to the complete suite and also to the position of the test case in the suite. The relation by which the passed cases are given reward can be tweaked and experimented
	## for making a more efficient reward function

	#rewards = np.cumsum(rewards)  # Reward for passed test cases
	#print(rewards)
	rewards[rank_idx] = total  # Rewards for failed testcases
	#print(rewards)

	ordered_rewards = []

	for tc in sc.testcases():
		try:
			idx = sc.scheduled_testcases.index(tc)  # Slow call
			#print(idx)
			ordered_rewards.append(rewards[idx])
		except ValueError:
			ordered_rewards.append(0.0)  # Unscheduled test case

	#print('timerank : ', len(ordered_rewards), ordered_rewards)
	return ordered_rewards

def timerank(result, sc):
	if result[0] == 0:
		return 0.0


	total = result[0]
	#print(total)

	## Detection Ranks are the positions of the failed test cases after our algorithm has prioritized them.
	## 실패할꺼라고 예상 한거중에서 실제로 실패한 TC의 index가 rank_idx인듯
	rank_idx = np.array(result[-2])-1
	#print(rank_idx)
	## no_scheduled is the total no of failed test cases scheduled by the algorithm
	## 실패할꺼라고 예상한 test case들의 sequence의 길이가 no_scheduled인듯
	no_scheduled = len(sc.scheduled_testcases)
	#print('no_scheduled : ', no_scheduled)
	rewards = np.zeros(no_scheduled)
	#print(rewards)

	## The rewards for the test cases that are failing is 1
	rewards[rank_idx] = 1
	#print(rewards)

	## The time rank gives reward to the complete suite and also to the position of the test case in the suite. The relation by which the passed cases are given reward can be tweaked and experimented 
	## for making a more efficient reward function

	rewards = np.cumsum(rewards)  # Reward for passed test cases
	#print(rewards)
	rewards[rank_idx] = total  # Rewards for failed testcases
	#print(rewards)

	ordered_rewards = []

	for tc in sc.testcases():
		try:
			idx = sc.scheduled_testcases.index(tc)  # Slow call
			#print(idx)
			ordered_rewards.append(rewards[idx])
		except ValueError:
			ordered_rewards.append(0.0)  # Unscheduled test case

	#print('timerank : ', len(ordered_rewards), ordered_rewards)
	return ordered_rewards


def newtcfail(result, sc):
	if result[0] == 0:
		# print(0)
		return 0.0

	total = result[0]
	rank_idx = np.array(result[-2]) - 1
	no_scheduled = len(sc.scheduled_testcases)

	# No_Scheduled are the total no of test cases that are scheduled in a CI cycle
	rewards = np.zeros(no_scheduled)
	rewards[rank_idx] = total
	# print(len(rewards), rewards)

	ordered_rewards = []

	for tc in sc.testcases():
		try:
			idx = sc.scheduled_testcases.index(tc)
			ordered_rewards.append(rewards[idx])
		# print(ordered_rewards)
		except ValueError:
			ordered_rewards.append(0.0)  # Unscheduled test case #해당 index가 unscheduled라서 reward값이 없는 경우인듯?

	#print('newtcfail : ', len(ordered_rewards), ordered_rewards)
	return ordered_rewards

## The tcfail gives reward per test case which is equal to the verdict of the test case. For test cases not scheduled, the verdict is considered as passed.
## For test cases that have been scheduled and failed , it gives a reward of 1 and for all others it gives a reward of 0.
def tcfail(result, sc):

	# The tcfail gives reward per test case which is equal to the verdict of the test case. For test cases not scheduled, the verdict is considered as passed and thus 0.
	## If the test case has passed, the reward is 1, if it is scheduled and failed , then it is 1.
	if result[0] == 0:
		#print(0)
		return 0.0

	total = result[0]
	rank_idx = np.array(result[-2])-1
	no_scheduled = len(sc.scheduled_testcases) 
	
	#No_Scheduled are the total no of test cases that are scheduled in a CI cycle
	rewards = np.zeros(no_scheduled)
	rewards[rank_idx] = 1
	#print(len(rewards), rewards)

	ordered_rewards = []

	for tc in sc.testcases():
		try:
			idx = sc.scheduled_testcases.index(tc)
			ordered_rewards.append(rewards[idx])
			#print(ordered_rewards)
		except ValueError:
			ordered_rewards.append(0.0)  # Unscheduled test case #해당 index가 unscheduled라서 reward값이 없는 경우인듯?

	#print('tcfail : ', len(ordered_rewards), ordered_rewards)
	return ordered_rewards
