import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  corpus: PropTypes.string.isRequired,
  dimensionality: PropTypes.arrayOf(PropTypes.number).isRequired,
  vocabSize: PropTypes.number.isRequired,
  tokens: PropTypes.string.isRequired,
  plotSrc: PropTypes.string.isRequired,
});
