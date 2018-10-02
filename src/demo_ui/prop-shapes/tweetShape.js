import PropTypes from 'prop-types';

export default PropTypes.shape({
  tweetId: PropTypes.string.isRequired,
  username: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
});
