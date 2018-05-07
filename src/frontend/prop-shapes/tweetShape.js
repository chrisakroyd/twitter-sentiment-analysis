import PropTypes from 'prop-types';

export default PropTypes.shape({
  tweetId: PropTypes.number.isRequired,
  username: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
});
