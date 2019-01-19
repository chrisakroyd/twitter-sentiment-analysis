import React from 'react';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';
import thunk from 'redux-thunk';

import { getExample } from './actions/compositeActions';

import sentimentDemo from './reducers';
import Root from './components/Root';

import './index.scss';
import createHashHistory from 'history/createHashHistory';

let middleware;

const history = createHashHistory();


if (process.env.NODE_ENV !== 'production') {
  const { logger } = require('redux-logger');
  const { composeWithDevTools } = require('redux-devtools-extension');
  middleware = composeWithDevTools(applyMiddleware(
    routerMiddleware(history),
    thunk,
    logger,
  ));
} else {
  middleware = applyMiddleware(thunk);
}

const reducers = combineReducers({
  ...sentimentDemo,
  router: routerReducer,
});

const defaultState = {
  predictions: {
    tokens: ['Hey', '<user>', 'I', 'like', 'you', ',', 'check', 'this', 'out', '<url>'],
    enabled: new Array(10).fill(true),
    attentionWeights: [0.046337228268384933, 0.05221894010901451, 0.0850118100643158, 0.09090767055749893, 0.07975268363952637, 0.05724440887570381, 0.03628542646765709, 0.047189291566610336, 0.06301657110452652, 0.03931921720504761, 0.021808134391903877],
    label: 'Positive',
    probs: [0.89, 0.06, 0.05],
    loading: false,
    error: null,
  },
  text: {
    text: 'Hey @_ChrisAkroyd_, I like you, check this out http://bbc.co.uk',
    loading: false,
    error: null,
  },
};

const store = createStore(
  reducers,
  defaultState,
  middleware,
);

// store.dispatch(getExample());

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);
