import React from 'react';
import './train.scss';

const Train = ({}) => (
  <div className="dash-body train">
    <div className="body-header">
      <h1>Train</h1>
    </div>
    <div className="body">
      <div className="train-error">
        <i className="material-icons">warning</i>
        <h2>Critical Error: Could not connect to cloud GPU Instance.</h2>
        <h2>Remote training unavailable.</h2>
      </div>
    </div>
  </div>
);

export default Train;
