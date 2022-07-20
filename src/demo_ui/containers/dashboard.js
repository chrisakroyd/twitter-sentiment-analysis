import { connect } from 'react-redux';

// Actions
import { getPrediction, predictWithTokens, getExample, setText } from '../actions/compositeActions';
import { toggleToken } from '../actions/predictActions';
import Demo from '../components/Demo';


const mapStateToProps = state =>
  ({
    text: state.text,
    predictions: state.predictions,
  });

const mapDispatchToProps = dispatch => ({
  loadExample: () => {
    dispatch(getExample());
  },
  process: () => {
    console.log('process');
    dispatch(getPrediction());
  },
  toggleToken: (index) => {
    dispatch(toggleToken(index));
    dispatch(predictWithTokens());
  },
  setText: (text) => {
    dispatch(setText(text));
  },
});

export default connect(mapStateToProps, mapDispatchToProps)(Demo);
