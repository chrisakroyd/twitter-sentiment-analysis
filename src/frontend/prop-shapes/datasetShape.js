import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  bestModel: PropTypes.string.isRequired,
  size: PropTypes.number.isRequired,
  words: PropTypes.number.isRequired,
  bestF1Score: PropTypes.number.isRequired,
  statistics: PropTypes.arrayOf(PropTypes.shape({
    name: PropTypes.string.isRequired,
    count: PropTypes.number.isRequired,
  })),
});
