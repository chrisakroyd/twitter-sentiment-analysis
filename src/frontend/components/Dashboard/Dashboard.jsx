import React from 'react';
import PropTypes from 'prop-types';
import { Route } from 'react-router';

import Sidebar from '../../components/Dashboard/Sidebar/Sidebar';

import Live from '../../components/Dashboard/Live/Live';
import Data from '../../components/Dashboard/Data/Data';
import Models from '../../components/Dashboard/Models/Models';
import Train from '../../components/Dashboard/Train/Train';
import Results from '../../components/Dashboard/Results/Results';

import genSidebarDescriptors from './genSidebarDescriptors';

import './dashboard.scss';

class Dashboard extends React.Component {
  render() {
    return (
      <div className="dashboard">
        <Sidebar
          highlightedItem={this.props.activeView}
          links={genSidebarDescriptors(this.props.match)}
          onSidebarLink={this.props.onDashClick}
        />
        <div>
          <Route
            exact
            path="/"
            render={() => (
              <Live
                status={this.props.status}
                activeText={this.props.activeText}
                process={this.props.process}
                setText={this.props.setText}
                tweets={this.props.tweets}
              />
            )}
          />

          <Route
            exact
            path="/data"
            render={() => (
              <Data />
            )}
          />

          <Route
            exact
            path="/models"
            render={() => (
              <Models />
            )}
          />

          <Route
            exact
            path="/train"
            render={() => (
              <Train />
            )}
          />

          <Route
            exact
            path="/results"
            render={() => (
              <Results />
            )}
          />
        </div>
      </div>
    );
  }
}

Dashboard.propTypes = {
  // Functions
  onDashClick: PropTypes.func.isRequired,
  process: PropTypes.func.isRequired,
  setText: PropTypes.func.isRequired,
  // Data
  loading: PropTypes.bool.isRequired,
  activeView: PropTypes.string.isRequired,
  activeText: PropTypes.shape({}).isRequired,
  embeddings: PropTypes.shape({}).isRequired,
  datasets: PropTypes.shape({}).isRequired,
  models: PropTypes.shape({}).isRequired,
  results: PropTypes.shape({}).isRequired,
  tweets: PropTypes.shape({}).isRequired,
  status: PropTypes.shape({}).isRequired,
};


export default Dashboard;