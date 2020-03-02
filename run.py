import sys


if sys.argv[1].lower() == 'prepfortrain':
    import PrepForTrain
    PrepForTrain.Prep()
if sys.argv[1].lower() == 'trainer':
    import trainer
    # improve here
if sys.argv[1].lower() == 'prediction':
    import predictor
    predictor.Predictor().tell(sys.argv[2])

