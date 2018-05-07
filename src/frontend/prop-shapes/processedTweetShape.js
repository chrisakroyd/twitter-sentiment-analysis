import PropTypes from 'prop-types';

export default PropTypes.shape({
  tweetId: PropTypes.number.isRequired,
  username: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
  tokenized: PropTypes.string.isRequired,
  processed: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
});
