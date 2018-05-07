import PropTypes from 'prop-types';

export default PropTypes.shape({
  name: PropTypes.string.isRequired,
  rows: PropTypes.number.isRequired,
  labels: PropTypes.arrayOf(PropTypes.string),
});
