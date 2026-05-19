// calibrate_presets.js — run from project root after server.js is running.
import fetch from "node-fetch";

// These preset names match the class ids used by the trained Vigil AI model.
const PRESETS = ["benign","ddos","dos","portscan","botnet","infiltration","webattack"];
const EXPECTED = {benign:0, ddos:1, dos:2, portscan:3, botnet:4, infiltration:5, webattack:6};

async function callPredict(features){
  // Calibration goes through HTTP so it tests the same path the web UI uses.
  const r = await fetch("http://localhost:3000/api/predict", {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({features})
  });
  return r.json();
}

async function run(){
  // Try each preset at increasing intensities to see when the model starts
  // matching the intended class often enough for demo examples.
  for(const preset of PRESETS){
    for(let intensity=0.5; intensity<=3.0; intensity+=0.5){
      let hits=0, trials=20;
      for(let t=0;t<trials;t++){
        // Create features by requesting the frontend generation endpoint? If not available, read exported vectors
        // For simplicity, generate random high spikes for amplify indices here:
        const features = Array(32).fill(0).map(()=>Math.random()*0.5);
        // Amplify feature groups that roughly line up with each traffic type.
        if (preset==="ddos"){ [10,11,2,4,24,30].forEach(i=>features[i-1]+=Math.random()*intensity*10); }
        if (preset==="dos"){ [11,10,3,30,12].forEach(i=>features[i-1]+=Math.random()*intensity*8); }
        if (preset==="portscan"){ [2,3,24,26].forEach(i=>features[i-1]+=Math.random()*intensity*6); }
        if (preset==="botnet"){ [11,4,13,27].forEach(i=>features[i-1]+=Math.random()*intensity*6); }
        if (preset==="infiltration"){ [1,11,24,26].forEach(i=>features[i-1]+=Math.random()*intensity*5); }
        if (preset==="webattack"){ [10,11,5,13].forEach(i=>features[i-1]+=Math.random()*intensity*6); }

        const resp = await callPredict(features);
        if (resp && typeof resp.prediction!=="undefined" && Number(resp.prediction)===EXPECTED[preset]) hits++;
      }
      console.log(preset, intensity.toFixed(1), "=>", (hits/20).toFixed(2));
    }
  }
}
run();
