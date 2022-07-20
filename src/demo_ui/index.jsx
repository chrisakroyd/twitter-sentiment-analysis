import React from 'react';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import thunk from 'redux-thunk';

import { loadClasses } from './actions/compositeActions';
import sentimentDemo from './reducers';
import Root from './components/Root';

import './index.scss';
let middleware;

if (process.env.NODE_ENV !== 'production') {
  const { logger } = require('redux-logger');
  const { composeWithDevTools } = require('redux-devtools-extension');
  middleware = composeWithDevTools(applyMiddleware(
    thunk,
    logger,
  ));
} else {
  middleware = applyMiddleware(thunk);
}

const reducers = combineReducers({
  ...sentimentDemo,
});

const defaultState = {
  predictions: {
    tokens: ['Hey', ',', 'I', 'love', 'this', 'website', ',', 'check', 'it', 'out', '<url>'],
    enabled: new Array(11).fill(true),
    attentionWeights: [0.046337228268384933, 0.05221894010901451, 0.0850118100643158, 0.09090767055749893, 0.07975268363952637, 0.05724440887570381, 0.03628542646765709, 0.047189291566610336, 0.06301657110452652, 0.03931921720504761, 0.021808134391903877],
    label: 'Positive',
    probs: [0.49, 0.06, 0.45],
    loading: false,
    error: null,
  },
  text: {
    text: 'Hey, I love this website, check it out http://chrisakroyd.com',
    classes: { 0: 'neutral', 1: 'negative', 2: 'positive' },
    loading: false,
    error: null,
  },
};

const store = createStore(
  reducers,
  defaultState,
  middleware,
);

store.dispatch(loadClasses());

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);
