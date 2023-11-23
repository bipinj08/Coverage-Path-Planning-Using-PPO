import copy
import tqdm
import distutils.util



from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay
from src.base.GridActions import GridActions


Final_training_results = []
Final_test_results = []

class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.agent = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0

        #self.total_rewards = []

    def train_episode(self):
        print('Training started')
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step(state)
            self.trainer.train_agent()

        #train_results = self.stats.final_callbacks_value()
        # for key, value in train_results:
        #     bound_method = value
        #     Unbounded = bound_method()
        #     each_episode_result = [key, Unbounded]
        #     Final_training_results.append(each_episode_result)


        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def run(self):
        print(type(self.agent))
        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

        # Training_file = self.generating_csv_file(Final_training_results, 'Training data')
        # Testing_file = self.generating_csv_file(Final_test_results, 'Testing data')
        #
        # Training_result_graph = self.Graph_visualization('Training')
        # Testing_result_graph = self.Graph_visualization('Test')

        #print(self.total_rewards)


    def step(self, state):
        old_state = copy.deepcopy(state)
        action, action_onehot, prediction = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))
        reward = self.rewards.calculate_reward(old_state, GridActions(action), next_state)
        self.trainer.add_experience(state, action_onehot, reward, next_state, prediction)
        self.stats.add_experience((old_state, action, reward, copy.deepcopy(next_state)))
        self.step_count += 1
        return copy.deepcopy(next_state)


    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def test_episode(self, scenario=None):
        print('Testing started')
        state = copy.deepcopy(self.init_episode(scenario))
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            action, action_onehot, prediction = self.agent.act(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)


        # test_results = self.stats.final_callbacks_value()
        # for key, value in test_results:
        #     bound_method = value
        #     Unbounded = bound_method()
        #     each_episode_result = [key, Unbounded]
        #     Final_test_results.append(each_episode_result)
        #     #print(Final_test_results)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step = self.step_count)


    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass

