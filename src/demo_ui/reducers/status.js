import { STATUS, STATUS_SUCCESS, STATUS_FAILURE } from '../constants/actions';

const usageMin = 2.4;
const usageMax = 2.8;

const loadMin = 0.01;
const loadMax = 8.9;

const status = (state = {}, action) => {
  switch (action.type) {
    case STATUS:
      return {
        connected: false,
        graphicsCard: 'Geforce GTX 1050',
        load: Math.random() * (loadMax - loadMin) + loadMin,
        memoryUsage: Math.random() * (usageMax - usageMin) + usageMin,
        maxMemoryUsage: 4.0,
        loading: false,
      };
    case STATUS_SUCCESS:
      console.log(action);
      return {
        connected: action.status[0].connected,
        graphicsCard: 'Geforce GTX 1050',
        load: Math.random() * (loadMax - loadMin) + loadMin,
        memoryUsage: Math.random() * (usageMax - usageMin) + usageMin,
        maxMemoryUsage: 4.0,
        loading: false,
      };
    case STATUS_FAILURE:
      return {
        connected: false,
        graphicsCard: '',
        load: 0.0,
        memoryUsage: 0.0,
        maxMemoryUsage: 4.0,
        loading: false,
      };
    default:
      return state;
  }
};

export default status;
