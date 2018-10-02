import React from 'react';
import PropTypes from 'prop-types';
import './refresh-button.scss';

const RefreshButton = ({ onRefresh }) => (
  <div className="refresh-button" onClick={() => onRefresh()}>
    <i className="material-icons">cached</i>
  </div>
);

RefreshButton.propTypes = {
  onRefresh: PropTypes.func.isRequired,
};

export default RefreshButton;
