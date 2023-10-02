import React, { useState, useEffect } from "react";
import Api from "./Api";
import "./normalize.css";
import "./default.css";
import "./main.css";
import Navbar from "./components/Navbar";
import Module from "./components/Module";
import Placeholder from "./images/no_image_placeholder.jpg";

function App() {
  const [livedata, setLivedata] = useState({ cam: null, modules: [] });

  useEffect(() => {
    setInterval(() => {
      Api.getLiveData()
        .then((res) => {
          setLivedata(res.data);
        })
        .catch((error) => {
          console.log("Live data fetch error");
        });
    }, 1000);
  }, []);

  return (
    <>
      <div className="body-background"></div>
      <div className="page">
        <Navbar />
        <section className="headings-section">
          <div className="cam-feed-container">
            <h3>Camera Live Feed</h3>
            <div className="card">
              <img
                src={
                  livedata.cam
                    ? `data:image/png;base64,${livedata.cam.img}`
                    : Placeholder
                }
                alt="Graphite Texture"
                className="cam-feed-img"
              />
            </div>
          </div>
          {livedata.modules.map((module) => {
            return <Module key={module.module} module={module} />;
          })}
        </section>
        {/* <section className="section-for-card">
          <article className="card my-example-card">
            <h1>Heading 1</h1>
            <h2>Heading 2</h2>
            <h3>Heading 3</h3>
            <h4>Heading 4</h4>
            <h5>Heading 5</h5>
            <h6>Heading 6</h6>
            <p>
              Paragraph: Lorem ipsum dolor, sit amet consectetur adipisicing
              elit. Corporis at adipisci nulla blanditiis facere unde eos odit
              doloribus explicabo doloremque! Possimus rem sunt quas tempora
              illum voluptatem officiis ex distinctio.
            </p>
            <p className="unimportant-text">
              Unimportant Text: Lorem ipsum dolor sit amet consectetur
              adipisicing elit. Nihil, maiores.
            </p>
            <div className="card-section section-for-card ">
              <button className="btn">Normal</button>
              <button className="btn btn-blue">Blue</button>
              <button className="btn btn-red">Red</button>
              <button className="btn btn-orange">Orange</button>
              <button className="btn btn-green">Green</button>
            </div>
            <h4 style={{ marginTop: "10px" }}>List of Items</h4>
            <ul>
              <li>First Item</li>
              <li>Second Item</li>
              <li>Third Item</li>
            </ul>
          </article>
        </section>
        <section className="section-for-card">
          <article className="card hoverable-card my-example-card">
            <h2>Hoverable Card</h2>
            <img src={Image} alt="Graphite Texture" />
            <h3>Lorem ipsum dolor sit amet.</h3>
            <p className="unimportant-text">Lorem ipsum dolor sit amet.</p>
            <p>
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
              Aspernatur, odio.
            </p>
          </article>
        </section>
        <section className="section-for-card">
          <form className="card my-form">
            <h3>Form Example</h3>
            <div className="notification notification-success">Successful</div>
            <div className="notification notification-danger">Denied</div>
            <div className="form-field-container">
              <label for="name" class="form-label">
                Name :
              </label>
              <input
                type="text"
                placeholder="Enter Name"
                id="name"
                class="form-input"
              />

              <label for="number" class="form-label">
                Number :
              </label>
              <input
                type="number"
                id="number"
                class="form-input"
                placeholder="Enter Phone Number"
              />

              <label for="email" class="form-label">
                Email :
              </label>
              <div className="form-input-alert-container">
                <input
                  type="email"
                  id="email"
                  class="form-input form-focus"
                  placeholder="Enter Email"
                  autoFocus
                />
                <small class="form-alert">Please Provide Value</small>
              </div>

              <label for="textarea" class="form-label">
                Textarea :
              </label>
              <textarea
                class="form-textarea"
                placeholder="Some long value"
              ></textarea>
              <button className="btn btn-red" disabled>
                Cancel
              </button>
              <button className="btn btn-green">Submit</button>
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "row",
                columnGap: "1rem",
                alignItems: "center",
              }}
            >
              <div className="loader loader-bg"></div>
              <div className="loader"></div>
              <div className="loader loader-sm"></div>
            </div>
          </form>
        </section>*/}
      </div>
    </>
  );
}

export default App;
