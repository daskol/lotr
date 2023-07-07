from lotr_glue import train


def test_train():
    train(task='cola',
          batch_size=8,
          num_epoches=1,
          enable_lotr=True,
          rank=2,
          lr=1e-3)
