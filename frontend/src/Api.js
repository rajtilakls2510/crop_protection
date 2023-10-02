import axios from "axios";

const API_URL = "http://192.168.83.96:5000";

const getLiveData = () => {
  return axios.get(API_URL + "/live");
};

const postManual = (module, override) => {
  var urlEncodedForm = new URLSearchParams();
  urlEncodedForm.append("module", module);
  urlEncodedForm.append("override", override);

  return axios.post(API_URL + `/override`, urlEncodedForm, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
};

const postStart = (module, control) => {
  var urlEncodedForm = new URLSearchParams();
  urlEncodedForm.append("module", module);
  urlEncodedForm.append("control", control);

  return axios.post(API_URL + `/control`, urlEncodedForm, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
};

const apiFunctions = {
  getLiveData,
  postManual,
  postStart,
};
export default apiFunctions;
