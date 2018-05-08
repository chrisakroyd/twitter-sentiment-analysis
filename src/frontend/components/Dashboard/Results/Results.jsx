import React from 'react';
import PropTypes from 'prop-types';
import shortId from 'shortid';
import ResultTile from './ResultTile/ResultTile';
import resultShape from '../../../prop-shapes/resultShape';

class Results extends React.Component {
  createTiles() {
    return this.props.results.map(data => (
      <ResultTile
        key={shortId.generate()}
        result={data}
      />
    ));
  }

  render() {
    return (
      <div className="dash-body">
        <div className="body-header">
          <h1>Results</h1>
        </div>
        <div className="tile-row">
          {this.createTiles()}
        </div>
      </div>
    );
  }
}

Results.propTypes = {
  results: PropTypes.arrayOf(resultShape).isRequired,
};

export default Results;
