import { connect } from 'react-redux';
import { push } from 'react-router-redux';
import { withRouter } from 'react-router-dom';

// Actions
import { getPrediction, getTweets } from '../actions/compositeActions';
import { changeDashView } from '../actions/dashboardActions';
import { setInputText } from '../actions/textInputActions';
import Dashboard from '../components/Dashboard/Dashboard';


const mapStateToProps = state =>
  ({
    activeView: state.appState.activeView,
    tweets: state.tweets,
    activeText: state.activeText,
    embeddings: state.embeddings,
    datasets: state.datasets,
    models: state.models,
    results: state.results,
    status: state.status,
  });

const mapDispatchToProps = dispatch => ({
  changeDashView: (type) => {
    dispatch(changeDashView(type));
  },
  onDashClick: (label, type, newUrl) => {
    console.log(newUrl);
    dispatch(changeDashView(type));
    dispatch(push(newUrl));
  },
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

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Dashboard));
