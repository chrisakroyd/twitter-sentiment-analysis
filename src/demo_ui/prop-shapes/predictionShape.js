import PropTypes from 'prop-types';
import errorShape from './errorShape';

export default PropTypes.shape({
  tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  enabled: PropTypes.arrayOf(PropTypes.bool).isRequired,
  attentionWeights: PropTypes.arrayOf(PropTypes.number).isRequired,
  probs: PropTypes.arrayOf(PropTypes.number).isRequired,
  label: PropTypes.string.isRequired,
  error: errorShape,
});
