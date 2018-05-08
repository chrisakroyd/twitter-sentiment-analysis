import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  size: PropTypes.string.isRequired,
  lastModified: PropTypes.string.isRequired,
  timePerEpoch: PropTypes.number.isRequired,
  learnRate: PropTypes.number.isRequired,
  clipNorm: PropTypes.number.isRequired,
  optimizer: PropTypes.string.isRequired,
  embedDrop: PropTypes.number.isRequired,
  noise: PropTypes.number.isRequired,
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
    topK: PropTypes.number.isRequired,
    classes: PropTypes.number.isRequired,
  }).isRequired,
});
