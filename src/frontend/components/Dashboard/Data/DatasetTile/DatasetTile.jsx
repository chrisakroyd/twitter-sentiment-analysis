import React from 'react';
import PropTypes from 'prop-types';

import './dataset-tile.scss';
import DonutChart from '../DonutChart/DonutChart';

const DatasetTile = ({ name, size, words, bestModel, bestF1Score, statistics }) =>
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
          <DonutChart statistics={statistics} />
        </div>
      </div>
    </div>
  );

DatasetTile.propTypes = {
  name: PropTypes.string.isRequired,
  size: PropTypes.number.isRequired,
  words: PropTypes.number.isRequired,
  bestModel: PropTypes.string.isRequired,
  bestF1Score: PropTypes.number.isRequired,
  statistics: PropTypes.arrayOf(PropTypes.shape({
    name: PropTypes.string.isRequired,
    count: PropTypes.number.isRequired,
  })).isRequired,
};

export default DatasetTile;
