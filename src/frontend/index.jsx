import React from 'react';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';
import thunk from 'redux-thunk';

import { getTweets } from './actions/compositeActions';

import searchApp from './reducers';
import Root from './components/Root';
import { SIDEBAR_LIVE } from './constants/sidebar';

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
    tweets: [
      {
        tweetId: 620013074272137216,
        username: 'UNKNOWN',
        text: 'Call for reservations for lunch or dinner tomorrow (yep Sunday!). Happy to accommodate guests in town for the MISS USA Pageant 346-5100',
      }, {
        tweetId: 620380830989336576,
        username: 'UNKNOWN',
        text: 'Tune in to the 2015 MISS USA Pageant July 12 8p ET/5p PT in Baton Rouge LIVE on ReelzChannel',
      }, {
        tweetId: 678053702306013185,
        username: 'UNKNOWN',
        text: '@AdamPlatt1999 one-sided support for Russia, which the Iran deal may\'ve began addressing. 2. Not doing so to a certain extent would lead to',
      }, {
        tweetId: 678420166221234176,
        username: 'UNKNOWN',
        text: '@RYOmoha @PrinceShaarawy ur fact: Milan hv a good squad and in the real world fact: Milan is 8th Placer hehe',
      }, {
        tweetId: 678437570506792960,
        username: 'UNKNOWN',
        text: '@CaptainLauren48 For your 21st, I got you a date with Chris Evans.  Sound good?  lol  :P',
      },
    ],
    loading: false,
  },
  status: {
    connected: true,
    graphicsCard: 'Geforce GTX 1080 TI',
    load: 5.4,
    memoryUsage: 4.2,
    maxMemoryUsage: 11.0,
    loading: false,
  },
  embeddings: {
    embeddings: [],
    loading: false,
  },
  datasets: {
    datasets: [],
    loading: false,
  },
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
