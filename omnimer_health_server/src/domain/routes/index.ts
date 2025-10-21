import { Express } from "express";
import permissionRoute from "./permission.route";
import roleRoute from "./role.route";
import authRoute from "./auth.route";

function setupRoutes(app: Express) {
  app.use("/api/v1/permission", permissionRoute);
  app.use("/api/v1/role", roleRoute);
  app.use("/api/v1/auth", authRoute);
}
export default setupRoutes;
