import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';

// Actions
import { getPrediction, getTweets, setText } from '../actions/compositeActions';
import Demo from '../components/Demo';


const mapStateToProps = state =>
  ({
    tweets: state.tweets,
    predictions: state.predictions,
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
    dispatch(setText(text));
  },
});

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Demo));
