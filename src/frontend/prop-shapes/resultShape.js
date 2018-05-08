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
  trainF1Improvement: PropTypes.number.isRequired,
  trainAccImprovement: PropTypes.number.isRequired,
  valF1Improvement: PropTypes.number.isRequired,
  valAccImprovement: PropTypes.number.isRequired,
});
