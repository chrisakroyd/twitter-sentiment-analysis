import React from 'react';

import './model-tile.scss';
import modelShape from '../../../../prop-shapes/modelShape';

const ModelTile = ({ model }) =>
  (
    <div className="model-tile tile">
      <div className="tile-header">
        <h3>{model.name} - {model.lastModified}</h3>
      </div>
      <div className="tile-body">
        <div className="model-visualiser">
          <h4>{model.name} Neural Visualisation</h4>
          <div className="visualisation">

          </div>
        </div>
        <div className="result-stats">
          <h4>Parameters</h4>
          <p>{model.size}</p>
          <h4>Time Per Epoch</h4>
          <p>{model.timePerEpoch}</p>
          <h4>Optimizer</h4>
          <p>{model.optimizer}</p>
          <h4>Learn Rate / Clip Norm / Noise / Dropout</h4>
          <p>{model.learnRate} / {model.clipNorm}% / {model.noise} / {model.embedDrop}</p>
        </div>
      </div>
    </div>
  );

ModelTile.propTypes = {
  model: modelShape.isRequired,
};

export default ModelTile;
