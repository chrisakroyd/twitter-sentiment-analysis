import PropTypes from 'prop-types';

export default PropTypes.shape({
  errorCode: PropTypes.number.isRequired,
  errorMessage: PropTypes.string.isRequired,
  parameters: PropTypes.shape({
    text: PropTypes.string,
    tokens: PropTypes.arrayOf(PropTypes.string),
  }),
});
