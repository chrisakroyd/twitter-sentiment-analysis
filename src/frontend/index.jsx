import React from 'react';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';
import thunk from 'redux-thunk';

import { getTweets } from './actions/compositeActions';

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
    originalText: 'Hey @_ChrisAkroyd_, I like you, check this out http://bbc.co.uk',
    text: 'Hey @_ChrisAkroyd_, I like you, check this out http://bbc.co.uk',
    processed: 'Hey <user> , I like you , check this out <url>',
    attentionWeights: [0.15, 0.7, 0.01, 0.6, 0.9, 0.7, 0.01, 0.25, 0.3, 0.2, 0.4],
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

// store.dispatch(getTweets());

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);
