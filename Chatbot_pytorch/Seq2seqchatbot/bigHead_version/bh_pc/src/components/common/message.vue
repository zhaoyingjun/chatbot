<template>
    <JwChat scrollType="scroll" :taleList="list" @enter="bindEnter" v-model="inputMsg" :toolConfig="tool"
        :quickList="quickList">
    </JwChat>
</template>
<script>
export default {
    name: 'message',
    data() {
        return {
            inputMsg: '',
            list: [],
            tool: {
                callback: this.toolEvent
            },
            quickList: [
                { text: '这里是jwchat，您想了解什么问题。' },
                { text: 'jwchat是最好的聊天组件' },
                { text: '谁将烟焚散，散了纵横的牵绊；听弦断，断那三千痴缠。' },
                { text: '长夏逝去。山野间的初秋悄然涉足。' },
                { text: '江南风骨，天水成碧，天教心愿与身违。' },
                { text: '总在不经意的年生。回首彼岸。纵然发现光景绵长。' },
                { text: '外面的烟花奋力的燃着，屋里的人激情的说着情话' },
                { text: '假如你是云，我就是雨，一生相伴，风风雨雨；' },
                {
                    text: '即使泪水在眼中打转，我依旧可以笑的很美，这是你学不来的坚强。'
                },
                { text: ' 因为不知来生来世会不会遇到你，所以今生今世我会加倍爱你。' }
            ],
            userdata:{
                name:localStorage.nickname ? sessionStorage.nickname : '游客',
                avatar:localStorage.avatar ? sessionStorage.avatar : '../../static/image/default_avatar.png'
            },
            status:this.$route.params.status || false
        }
    },
    props: {
        resMsg: {
            type: Object,
            required: true
        }
    },
    watch: {
        resMsg: function (newVal, oldVal) {
            console.log(newVal);
            this.addMsg(newVal.answer)
        }
    },
    created(){
        this.chatMsg()
    },
    methods: {
        bindEnter(e) {
            console.log(this.status);
           if(this.status == false){
            console.log(e)
            const msg = this.inputMsg
            if (!msg) return
            this.$emit('sendMsg', msg)
            const msgObj = {
                date: this.dataTime(),
                text: { text: msg },
                mine: true,
                name: this.userdata.name,
                img: this.userdata.avatar
            }
            this.list.push(msgObj)
        }else{
            this.$message.warning('聊天记录仅支持查看')
        }
        },
        addMsg(val) {
            const msgObj = {
                date: this.dataTime(),
                text: { text: val },
                mine: false,
                name: '大头',
                img: '../../static/image/toubao.jpg'
            }
            this.list.push(msgObj)
        },
        dataTime() {
            //年
            let year = new Date().getFullYear()
            //月份是从0月开始获取的，所以要+1;
            let month = new Date().getMonth() + 1
            //日
            let day = new Date().getDate();
            //时
            let hour = new Date().getHours();
            //分
            let minute = new Date().getMinutes();
            //秒
            let second = new Date().getSeconds();
            let ymd = year + '-' + (month < 10 ? '0' + month : month)+'-'+(day < 10 ? '0' + day : day) + ' '
            let ms = (hour < 10 ? '0' + hour : hour) + ':' + (minute < 10 ? '0' + minute : minute) + ':' + (second < 10 ? '0' + second : second)
            return ms
            // console.log(time);
        },
        toolEvent(type, obj) {
            console.log('tools', type, obj)
        },
        chatMsg(){
            if( this.$route.params.chatMsg){
            const arr = []
            console.log(this.$route.params.chatMsg);
            this.$route.params.chatMsg.forEach(el=>{
                const personMsg = {
                    date:this.$moment(el.created_time).format('HH:mm:ss'),
                    text:{text:el.question},
                    mine:true,
                    name: this.userdata.name,
                    img:this.userdata.avatar
                }
                const msgObj = {
                date:this.$moment(el.created_time).format('HH:mm:ss'),
                text: { text: el.answer },
                mine: false,
                name: '大头',
                img: '../../static/image/toubao.jpg'
            }
                arr.push(personMsg)
                arr.push(msgObj)
            })
            console.log(this.list);
            this.$nextTick(()=>{
                this.list = arr
            })
        }
    }

    },
    mounted() {
        let dom = document.querySelector('.web__msg-input')
            console.log(dom);
            dom.onkeydown = function(event){
                let ev = event || window.event || arguments.callee.caller.arguments[0]
                console.log(ev);
                if(ev && ev.keyCode == 13){
                    ev.preventDefault()
                }
            }
        //  }
        const img = 'https://www.baidu.com/img/flexible/logo/pc/result.png'
        const list = [
        ]
        this.list = list
        this.dataTime()
    }
}
</script>
<style lang="less" scoped>
/deep/ .bscroll-indicator{
    border: none !important;
}
.chatPage {
    height: 100% !important;
    width: 100% !important;
    background: #44515a;
}

/deep/ .wrapper {
    width: 100% !important;
    height: 100% !important;
}

/deep/ .web__msg {
    display: flex;
    height: 30px;
    width: 100%;
    padding: 1px 10px 0px 10px;
}

/deep/ .web__msg-input {
    overflow-y: hidden;
    line-height: 35px;
    background: #2f3e49;
    color: #fff;
}

/deep/ .web__msg-menu {
    text-align: unset;

    .el-button {
        border: none;
        border-radius: 5px;
        background: #d4dce6;
        border-radius: 10px;

        position: absolute;
        z-index: 1;
        top: 8px;
        right: 10px;
        margin: 0 auto;
        padding: 6px 10px;

        color:#47555e;
        font-size: 10px;
        line-height: 1;
        text-align: center;
    }

    .el-button :hover{
        background: #e5e9ef;
    }
}

/deep/.taleBox {
    height: calc(100% - 45px) !important;
}

/deep/ .toolBox {
    display: flex;
    -webkit-box-flex: 0;
    -webkit-flex: 0 1 40px;
    -ms-flex: 0 1 40px;
    flex: 0 1 40px;
    width: 100%;
    background: #2f3e49;
}

/deep/ .toolBox {
    height: 40px !important;
}

/deep/ .toolIcon {
    margin: 8px 5px 2px 2px;
}

/deep/ .wrapper .web__main .web__main-arrow::after{
    content: " ";
    top: -14px;
    left: 1px;
    border-width: 0px;
}

/deep/ .web__main-text {
    border: none !important;
    background-color: #2d3a43 !important;
    color: #b7b5af !important;
    border-radius: 15px !important;
}

/deep/ .wrapper .web__main .web__main-item--mine .web__main-text{
    border-radius: 15px !important;
}

/deep/ .wrapper .web__main .web__main-item--mine .web__main-text .web__main-arrow::after{
    border-width: 0px;
}

/deep/ .wrapper .web__main .web__main-item--mine .web__main-text .web__main-arrow{
    right: 2px;
    left: 50px;
    top: 0px;
    border-color: transparent;
    border-style: solid;
    border-width: 0px;
}

/deep/ .web__msg-menu .el-button--primary :hover{
    background:none;
    color:#fff;
}
</style>
