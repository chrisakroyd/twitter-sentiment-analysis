import React from 'react';
import PropTypes from 'prop-types';
import { Provider } from 'react-redux';
import Home from '../containers/dashboard';


const Root = ({ store }) => (
  <Provider store={store}>
    <Home />
  </Provider>
);

Root.propTypes = {
  store: PropTypes.shape({}).isRequired,
  history: PropTypes.shape({}).isRequired,
};

export default Root;
