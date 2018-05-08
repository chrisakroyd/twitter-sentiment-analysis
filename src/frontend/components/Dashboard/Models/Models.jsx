import React from 'react';
import PropTypes from 'prop-types';
import shortId from 'shortid';

import './models.scss';
import ModelTile from './ModelTile/ModelTile';
import modelShape from '../../../prop-shapes/modelShape';

class Models extends React.Component {
  createTiles() {
    return this.props.models.map(data => (
      <ModelTile
        key={shortId.generate()}
        model={data}
      />
    ));
  }

  render() {
    return (
      <div className="dash-body">
        <div className="body-header">
          <h1>Models</h1>
        </div>
        <div className="tile-row">
          {this.createTiles()}
        </div>
      </div>
    );
  }
}

Models.propTypes = {
  models: PropTypes.arrayOf(modelShape).isRequired,
};


export default Models;
