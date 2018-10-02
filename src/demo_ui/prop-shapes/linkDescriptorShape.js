import PropTypes from 'prop-types';

export default PropTypes.shape({
  icon: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  type: PropTypes.string.isRequired,
  linkTo: PropTypes.string.isRequired,
});
