import Vue from 'vue'
import Router from 'vue-router'

import Index from "../components"
import Login from "../components/Login"
import Register from "../components/Register";
import User from "../components/User";
import Chat from "../components/Chat";
import Doc from "../components/Doc";
import Contact from "../components/Contact";
Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'Index',
      component: Index,
    },
    {
      name:"Login",
      path: "/user/login",
      component:Login,
    },
     {
      name:"Register",
      path: "/user/reg",
      component:Register,
    },
    {
      name:"User",
      path: "/user",
      component:User,
    },
    {
      name:"Chat",
      path: "/chat",
      component:Chat,
    },
    {
      name:"Doc",
      path: "/doc",
      component:Doc,
    },
    {
      name:"Contact",
      path: "/contact",
      component:Contact,
    },
  ]
})
