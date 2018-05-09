import React from 'react';
import PropTypes from 'prop-types';
import './search-button.scss';

class SearchButton extends React.Component {
  render() {
    return (
      <div className="search-button" onClick={() => this.props.onEnter()}>
        Process
      </div>
    );
  }
}

SearchButton.propTypes = {
  onEnter: PropTypes.func.isRequired,
};

export default SearchButton;
