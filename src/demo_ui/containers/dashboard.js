import { connect } from 'react-redux';
import { push } from 'react-router-redux';
import { withRouter } from 'react-router-dom';

// Actions
import { getPrediction, getTweets } from '../actions/compositeActions';
import { setInputText } from '../actions/textInputActions';
import Demo from '../components/Demo';


const mapStateToProps = state =>
  ({
    tweets: state.tweets,
    activeText: state.activeText,
    status: state.status,
  });

const mapDispatchToProps = dispatch => ({
  tweetRefresh: () => {
    dispatch(getTweets());
  },
  process: () => {
    console.log('process');
    dispatch(getPrediction());
  },
  setText: (text) => {
    dispatch(setInputText(text));
  },
});

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Demo));
