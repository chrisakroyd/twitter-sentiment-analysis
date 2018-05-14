import React from 'react';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';
import thunk from 'redux-thunk';

import { getTweets, getModelStatus } from './actions/compositeActions';

import searchApp from './reducers';
import Root from './components/Root';
import { SIDEBAR_LIVE } from './constants/sidebar';
import { tweets, status, datasets, embeddings, results, models } from './constants/defaults';


import './index.scss';
import createHashHistory from 'history/createHashHistory';

let middleware;

const history = createHashHistory();


if (process.env.NODE_ENV !== 'production') {
  const { logger } = require('redux-logger');
  const { composeWithDevTools } = require('redux-devtools-extension');
  middleware = composeWithDevTools(
    applyMiddleware(
      routerMiddleware(history),
      thunk,
      logger,
    ),
  );
} else {
  middleware = applyMiddleware(thunk);
}

const reducers = combineReducers({
  ...searchApp,
  router: routerReducer,
});

const defaultState = {
  appState: {
    activeView: SIDEBAR_LIVE,
  },
  activeText: {
    text: 'Hey @_ChrisAkroyd_, I like you, check this out http://bbc.co.uk',
    processed: 'Hey <user> , I like you , check this out <url>',
    attentionWeights: [0.046337228268384933, 0.05221894010901451, 0.0850118100643158, 0.09090767055749893, 0.07975268363952637, 0.05724440887570381, 0.03628542646765709, 0.047189291566610336, 0.06301657110452652, 0.03931921720504761, 0.021808134391903877],
    classification: 'Positive',
    confidence: 0.89,
    loading: false,
  },
  tweets: {
    loading: false,
    tweets,
  },
  embeddings: {
    loading: false,
    embeddings,
  },
  datasets: {
    loading: false,
    datasets,
  },
  results: {
    loading: false,
    results,
  },
  models: {
    loading: false,
    models,
  },
  status,
};

const store = createStore(
  reducers,
  defaultState,
  middleware,
);

store.dispatch(getTweets());
store.dispatch(getModelStatus());

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);
