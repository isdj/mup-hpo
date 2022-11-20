def hpt(args):
  from syne_tune.search_space import loguniform, uniform

  est = get_estimator(args)  # local sagemaker estimator creator
  max_epochs = 8
  for mup in [True, False]:
      for mult in range(1, 9):
          width = 192 * mult
          search_space = {"lr": loguniform(1e-6, 1e-1), "momentum": 0,
                          'epochs': max_epochs, 'dim': width}
          if mup:
              search_space['mup'] = ''

          from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler

          scheduler = HyperbandScheduler(
              search_space,
              max_t=max_epochs,
              resource_attr='epoch',
              searcher='random',
              metric="loss",
              mode="min",
              reduction_factor=2
          )
          from syne_tune.tuner import Tuner
          from syne_tune.stopping_criterion import StoppingCriterion
          from syne_tune.backend.sagemaker_backend.sagemaker_backend import SagemakerBackend
          tuner = Tuner(
              backend=SagemakerBackend(sm_estimator=est, inputs=inputs),
              scheduler=scheduler,
              stop_criterion=StoppingCriterion(max_num_trials_started=32),
              n_workers=8,
              tuner_name=f"segmenter-width-{width}-{'mup' if mup else 'no-mup'}"
          )
          tuner.run()

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    hpt(args)