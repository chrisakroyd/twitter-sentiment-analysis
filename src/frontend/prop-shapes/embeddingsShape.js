import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  vocabSize: PropTypes.number.isRequired,
  dimensionality: PropTypes.number.isRequired,
});
