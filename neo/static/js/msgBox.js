class MessageBox {
	constructor(options) {
		// 把传递进来的配置信息挂载到实例上（以后可以基于实例在各个方法各个地方拿到这个信息）
		for(let key in options) {
			if(!options.hasOwnProperty(key)) break;
			this[key] = options[key];
		}
		// 开始执行
		this.init();
	}
	// 初始化：通过执行INIT控制逻辑的进行
	init() {
		if(this.status === "message") {
			this.createMessage();
			this.open();
			return;
		}
	}
	// 创建元素
	createMessage() {
		this.messageBox = document.createElement('div');
		this.messageBox.className = `dpn-message dpn-${this.type}`;
		this.messageBox.innerHTML = `${this.message}<i class="dpn-close">X</i>`;
		document.body.appendChild(this.messageBox);
		// 基于事件委托监听关闭按钮的点击
		this.messageBox.onclick = ev => {
			let target = ev.target;
			//判断点击的元素是否为关闭按钮
			if(target.className === "dpn-close") {
				// 点击的是关闭按钮
				this.close();
			}
		};
		// 钩子函数
		this.oninit();
	}

	// 控制显示
	open() {
		if(this.status === "message") {
			let messageBoxs = document.querySelectorAll('.dpn-message'),len = messageBoxs.length;
			//计算新弹出的messageBox的Y轴偏移量
			this.messageBox.style.top = `${len===1 ? 20:20+(len-1)*70}px`;
			// 如果duration不为零，控制自动消失
			this.autoTimer = setTimeout(() => {
				this.close();
			}, this.duration);
			// 钩子函数
			this.onopen();
			return;
		}
	}
	// 控制隐藏
	close() {
		if(this.status === "message") {
			clearTimeout(this.autoTimer);
			this.messageBox.style.top = '-200px';
			let anonymous = () => {
				document.body.removeChild(this.messageBox);
				// 钩子函数
				this.onclose();
			};
			this.messageBox.addEventListener('transitionend', anonymous);
			return;
		}
	}
}

//全局对象上挂载该方法
window.messageplugin = function(options = {}) {
	//允许只传入字符串，对其进行对象格式处理
	if(typeof options === "string") {
		options = { message: options };
	}
	//用户提供的配置覆盖默认配置项
	options = Object.assign({
		status: 'message',
		message: '我是默认信息',
		type: 'info',
		duration: 1000,
		//生命周期钩子
		oninit() {},
		onopen() {},
		onclose() {},
	}, options);
	return new MessageBox(options);
};

/**************************************************  自定义confirm弹框  **************************************************/

let 自定义confirm弹框 = `
    <div id="confirm弹框" class="confirm弹框">
	    <div class="confirm-content">
		    <div class="confirm-message"></div>
		    <div class="confirm-buttons">
			    <button class="confirm-ok">确认</button>
			    <button class="confirm-cancel">取消</button>
		    </div>
	    </div>
    </div>`;

// 将confirm弹框添加到body中，默认样式是隐藏
document.body.insertAdjacentHTML('beforeend', 自定义confirm弹框);

const confirm弹框 = document.getElementById("confirm弹框");
const confirmMsg = confirm弹框.querySelector(".confirm-message");
const confirmOk = confirm弹框.querySelector(".confirm-ok");
const confirmCancel = confirm弹框.querySelector(".confirm-cancel");

/**
 * confirm弹框显示
 * */
function confirm弹框Show(msg) {
	// 将弹框显示出来
	confirm弹框.style.display = "block";
	confirmMsg.textContent = msg;
}

/**
 * 取消，关闭弹窗，不进行其他操作
 * */
confirmCancel.onclick = function (){
	confirm弹框.style.display = "none";
}
