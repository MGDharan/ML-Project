const express = require("express");
const http = require("http");
const bcrypt = require("bcryptjs");
const { randomBytes } = require("crypto");
const { MongoClient } = require("mongodb");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017";
const DB_NAME = process.env.DB_NAME || "quiz_battle";
const QUESTION_COUNT = 15;

app.use(express.json());
app.use(express.static("public"));

const socketsByPlayerId = {};
const games = {};

let db;
let usersCol;
let questionsCol;
let answersCol;
let gamesCol;

const seedQuestions = [
  { q: "Capital of India?", a: "Delhi" },
  { q: "5 + 5?", a: "10" },
  { q: "Sun rises from?", a: "East" },
  { q: "2 * 6?", a: "12" },
  { q: "Water formula?", a: "H2O" },
  { q: "Largest planet?", a: "Jupiter" },
  { q: "HTML stands for?", a: "Hypertext Markup Language" },
  { q: "CSS used for?", a: "Styling" },
  { q: "JS is?", a: "Programming Language" },
  { q: "Binary of 2?", a: "10" },
  { q: "7 + 8?", a: "15" },
  { q: "Opposite of hot?", a: "Cold" },
  { q: "Earth shape?", a: "Sphere" },
  { q: "Fastest land animal?", a: "Cheetah" },
  { q: "1 byte = ?", a: "8 bits" }
];

function normalizeAnswer(text) {
  return String(text || "").trim().toLowerCase();
}

function makePlayerId() {
  return `PLY-${randomBytes(4).toString("hex").toUpperCase()}`;
}

async function connectMongo() {
  const client = new MongoClient(MONGO_URI);
  await client.connect();
  db = client.db(DB_NAME);
  usersCol = db.collection("users");
  questionsCol = db.collection("questions");
  answersCol = db.collection("answers");
  gamesCol = db.collection("games");

  await usersCol.createIndex({ email: 1 }, { unique: true });
  await usersCol.createIndex({ playerId: 1 }, { unique: true });

  const questionCountInDb = await questionsCol.countDocuments();
  if (questionCountInDb === 0) {
    await questionsCol.insertMany(
      seedQuestions.map((item) => ({
        ...item,
        createdAt: new Date()
      }))
    );
  }

  console.log(`MongoDB connected: ${DB_NAME}`);
}

app.post("/api/register", async (req, res) => {
  try {
    const email = String(req.body.email || "").trim().toLowerCase();
    const password = String(req.body.password || "");

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters" });
    }

    const exists = await usersCol.findOne({ email });
    if (exists) {
      return res.status(409).json({ error: "Email already registered" });
    }

    let playerId = makePlayerId();
    while (await usersCol.findOne({ playerId })) {
      playerId = makePlayerId();
    }

    const passwordHash = await bcrypt.hash(password, 10);
    await usersCol.insertOne({
      playerId,
      email,
      passwordHash,
      createdAt: new Date(),
      lastLoginAt: new Date()
    });

    return res.json({ playerId, email });
  } catch (error) {
    console.error("Register error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/login", async (req, res) => {
  try {
    const email = String(req.body.email || "").trim().toLowerCase();
    const password = String(req.body.password || "");

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const user = await usersCol.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    await usersCol.updateOne(
      { _id: user._id },
      { $set: { lastLoginAt: new Date() } }
    );

    return res.json({
      playerId: user.playerId,
      email: user.email
    });
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/questions/count", async (_req, res) => {
  try {
    const count = await questionsCol.countDocuments();
    return res.json({ count: Math.min(count, QUESTION_COUNT) });
  } catch (error) {
    return res.status(500).json({ error: "Internal server error" });
  }
});

io.on("connection", async (socket) => {
  const authPlayerId = String(socket.handshake.auth?.playerId || "").trim();
  if (!authPlayerId) {
    socket.emit("authError", "Please login first");
    socket.disconnect(true);
    return;
  }

  const user = await usersCol.findOne({ playerId: authPlayerId });
  if (!user) {
    socket.emit("authError", "Invalid player session");
    socket.disconnect(true);
    return;
  }

  socket.data.playerId = user.playerId;
  socketsByPlayerId[user.playerId] = socket;

  socket.emit("yourId", user.playerId);

  socket.on("requestGame", (targetPlayerId) => {
    const target = socketsByPlayerId[targetPlayerId];
    if (target) {
      target.emit("gameRequest", user.playerId);
    }
  });

  socket.on("acceptGame", async (fromPlayerId) => {
    const requester = socketsByPlayerId[fromPlayerId];
    if (!requester) {
      return;
    }

    const gameId = `${fromPlayerId}-${user.playerId}-${Date.now()}`;
    const dbQuestions = await questionsCol
      .find({})
      .sort({ _id: 1 })
      .limit(QUESTION_COUNT)
      .toArray();

    if (dbQuestions.length === 0) {
      socket.emit("gameError", "No questions available in database");
      return;
    }

    games[gameId] = {
      players: [fromPlayerId, user.playerId],
      scores: { [fromPlayerId]: 0, [user.playerId]: 0 },
      currentQ: 0,
      answered: false,
      questions: dbQuestions
    };

    await gamesCol.insertOne({
      gameId,
      players: [fromPlayerId, user.playerId],
      startedAt: new Date(),
      status: "started"
    });

    requester.emit("startGame", gameId);
    socket.emit("startGame", gameId);

    const firstQuestion = dbQuestions[0]?.q || "";
    requester.emit("newQuestion", firstQuestion);
    socket.emit("newQuestion", firstQuestion);
  });

  socket.on("answer", async ({ gameId, answer }) => {
    const game = games[gameId];
    const playerId = socket.data.playerId;
    if (!game || game.answered || !game.players.includes(playerId)) {
      return;
    }

    const currentQuestion = game.questions[game.currentQ];
    if (!currentQuestion) {
      return;
    }

    const isCorrect =
      normalizeAnswer(answer) === normalizeAnswer(currentQuestion.a);

    await answersCol.insertOne({
      gameId,
      playerId,
      questionId: currentQuestion._id,
      question: currentQuestion.q,
      providedAnswer: String(answer || ""),
      correctAnswer: currentQuestion.a,
      isCorrect,
      createdAt: new Date()
    });

    if (isCorrect) {
      game.scores[playerId] += 1;
      game.answered = true;
    }

    const p1 = socketsByPlayerId[game.players[0]];
    const p2 = socketsByPlayerId[game.players[1]];

    if (p1) p1.emit("updateScore", game.scores);
    if (p2) p2.emit("updateScore", game.scores);

    setTimeout(async () => {
      game.currentQ += 1;
      game.answered = false;

      if (game.currentQ >= game.questions.length) {
        if (p1) p1.emit("gameOver", game.scores);
        if (p2) p2.emit("gameOver", game.scores);

        await gamesCol.updateOne(
          { gameId },
          {
            $set: {
              endedAt: new Date(),
              status: "finished",
              finalScores: game.scores
            }
          }
        );

        delete games[gameId];
      } else {
        const nextQ = game.questions[game.currentQ].q;
        if (p1) p1.emit("newQuestion", nextQ);
        if (p2) p2.emit("newQuestion", nextQ);
      }
    }, 1500);
  });

  socket.on("disconnect", () => {
    delete socketsByPlayerId[user.playerId];
  });
});

async function start() {
  try {
    await connectMongo();
    server.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error("Startup error:", error);
    process.exit(1);
  }
}

start();