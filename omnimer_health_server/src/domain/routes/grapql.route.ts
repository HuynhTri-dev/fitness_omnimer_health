import { Router } from "express";
import { GraqQLController } from "../controllers/GraqQL.controller";

const router = Router();
const graqQLController = new GraqQLController();

router.get("/user/:userId", graqQLController.getUserGraphData);

export default router;
