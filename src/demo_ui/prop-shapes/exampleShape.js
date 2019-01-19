import PropTypes from 'prop-types';

export default PropTypes.shape({
  text: PropTypes.string.isRequired,
  tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  num_tokens: PropTypes.number.isRequired,
  label: PropTypes.number.isRequired,
});
