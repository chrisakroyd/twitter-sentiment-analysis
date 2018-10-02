import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  lastRun: PropTypes.string.isRequired,
  lastModified: PropTypes.string.isRequired,
  runs: PropTypes.number.isRequired,
  trainF1: PropTypes.number.isRequired,
  valF1: PropTypes.number.isRequired,
  trainAcc: PropTypes.number.isRequired,
  valAcc: PropTypes.number.isRequired,
  trainLoss: PropTypes.number.isRequired,
  valLoss: PropTypes.number.isRequired,
  trainConfMatrixSrc: PropTypes.string.isRequired,
  valConfMatrixSrc: PropTypes.string.isRequired,
  tokenizationScheme: PropTypes.string.isRequired,
  trainF1Change: PropTypes.number.isRequired,
  trainAccChange: PropTypes.number.isRequired,
  trainLossChange: PropTypes.number.isRequired,
  valF1Change: PropTypes.number.isRequired,
  valAccChange: PropTypes.number.isRequired,
  valLossChange: PropTypes.number.isRequired,
});
