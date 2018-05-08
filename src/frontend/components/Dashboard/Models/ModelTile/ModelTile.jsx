import React from 'react';

import './model-tile.scss';
import modelShape from '../../../../prop-shapes/modelShape';

const ModelTile = ({ model }) =>
  (
    <div className="data-tile tile">
      <div className="tile-header">
        <h3>{name}</h3>
      </div>
      <div className="tile-body">
        <div className="data-stats">
          <h4>Dataset Size</h4>
          <p>{size}</p>
          <h4>Unique Words</h4>
          <p>{words}</p>
          <h4>Best Model on Dataset</h4>
          <p>{bestModel}</p>
          <h4>Highest Model F1-Score</h4>
          <p>{bestF1Score}%</p>
        </div>
        <div className="class-breakdown">
          <h4>Class Breakdown</h4>
        </div>
      </div>
    </div>
  );

ModelTile.propTypes = {
  model: modelShape.isRequired,
};

export default ModelTile;
