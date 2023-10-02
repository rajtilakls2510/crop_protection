import React, { useEffect, useState } from "react";
import moment from "moment";
import Api from "../Api";

const Module = ({ module }) => {
  const [manual, setManual] = useState(module.manual);
  const [alarm, setAlarm] = useState(module.severity > 0);
  useEffect(() => {
    setManual(module.manual);
    setAlarm(module.severity > 0);
  }, [module]);

  const getDateTime = (seconds) => {
    var t = new Date(seconds * 1000);
    var formatted = moment(t).format("hh:mmA DD.MM.YYYY");
    return formatted;
  };

  const onManualClick = () => {
    Api.postManual(module.module, !manual)
      .then((res) => {
        if (res.data === "Done") setManual(!manual);
      })
      .catch((error) => {
        console.log("Setting mode error");
      });
  };

  const onStartClick = () => {
    Api.postStart(module.module, alarm ? 0 : 1)
      .then((res) => {
        if (res.data === "Done") setAlarm(!alarm);
      })
      .catch((error) => {
        console.log("Setting alarm error");
      });
  };

  return (
    <div className="module-container">
      <h3>Module {module.module}</h3>
      <div className="card module-card-container">
        <div className="module-live-container">
          <h5>Motion:</h5>
          <span>{module.motion === 1 ? "DETECTED" : "NOT DETECTED"}</span>
          <h5>Distance: </h5>
          <span>{module.distance} cm</span>
        </div>
        <div className="card-section module-alarm-container">
          <div
            className={`notification ${
              alarm ? "notification-danger" : "notification-success"
            }`}
          >
            Alarm {alarm ? "ACTIVE" : "INACTIVE"}
          </div>
          <span>
            <button
              className="btn btn-blue"
              disabled={!manual}
              onClick={onStartClick}
            >
              {alarm ? "Stop" : "Start"}
            </button>
          </span>
          <span>
            <button className="btn btn-red" onClick={onManualClick}>
              {manual ? "Auto" : "Manual"}
            </button>
          </span>
        </div>
        <div className="card-section module-history-container">
          <h5>History:</h5>
          <div className="module-history-grid">
            <h6>Time:</h6>
            <h6>Severity:</h6>
            {module.history.reverse().map((his) => {
              return (
                <>
                  <span>{getDateTime(his.time)}</span>
                  <span>
                    {his.severity === 1
                      ? "Caution"
                      : his.severity === 2
                      ? "Warning"
                      : "Catastrophic"}
                  </span>
                </>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Module;
