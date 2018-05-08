import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  size: PropTypes.string.isRequired,
  timePerEpoch: PropTypes.number.isRequired,
  learningRate: PropTypes.number.isRequired,
  clipNorm: PropTypes.number.isRequired,
  topK: PropTypes.number.isRequired,
  optimizer: PropTypes.string.isRequired,
  bestAccuracy: PropTypes.number.isRequired,
  bestF1Score: PropTypes.number.isRequired,
  lastRun: '',
  classes: PropTypes.number.isRequired,
  architecture: PropTypes.shape({
    embeddingDim: PropTypes.number.isRequired,
    rnnUnits: PropTypes.number.isRequired,
    rnnLayers: PropTypes.number.isRequired,
    biDirectional: PropTypes.bool.isRequired,
    rnnParams: {
      l2: PropTypes.number.isRequired,
      dropout: PropTypes.number.isRequired,
    },
    attention: PropTypes.bool.isRequired,
    concPool: PropTypes.bool.isRequired,
    classes: PropTypes.number.isRequired,
  }).isRequired,
});
